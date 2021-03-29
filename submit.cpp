#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <fstream>
#include <string> 
#include <vector>
#include <math.h>
#include <iomanip>
#include <chrono>

#define DO_TIMING

using namespace std;

int id, p, tag_num = 3;
int rows, columns; //split the whole domain into grids
int id_row, id_column; //gird id for each processor
int grid_rows, grid_cols; //grid size for each processor
MPI_Datatype curr_col, ext_col; //create new datatype to send the column of the matrix

// divide P processors into a grid size of rows*columns, code from ex3 of worksheet2
void find_dimensions(int p,int id,int& rows, int& columns);
// find the index in the grid according to its id, code from ex3 of worksheet2
void id_to_index(int id, int& id_row, int& id_column);
// find its id according to its index in the grid, code from ex3 of worksheet2
int id_from_index(int id_row, int id_column);
// set the grid size for each processor
void set_grid(int& grid_rows, int rows_responsible_min_max[2]);
// randomly set per percentage of alive cells in each grid
void initialize(bool* current, double per);
void copygrid(bool* current, bool* grid_ext);
// determine the next life state inside grids
void determineState(bool* current, bool* grid_ext);
// calculate the grid size for each processor
void decomposition(int height, int rows, vector< vector<int> > &rows_responsible_list);
// send information to its neighbor grid
void send_neighbor( int i, int j, int com_id, bool* current, MPI_Request* request, int &cnt);
//receive information from its neighbor grid
void update_neighbor(int i, int j, int com_id, bool* grid_ext, MPI_Request* request, int &cnt);


int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	srand(time(NULL) + id * 10);

	//load command argument
    if (argc!=6)
    {
        cout<<"sorry, please input all necessary information in order: number of steps, boundary condition,width,height,number of alive cells"<<endl;
        exit(0);
    }
    
	int iterations = atoi(argv[1]);
	bool periodical = (atoi(argv[2]) == 0 ? false : true);
    int height = atoi(argv[3]);
	int width = atoi(argv[4]);
    double live_cells = atof(argv[5]); //how many live cells in the domain
    if(id==0){
        cout<<"domain size is:"<<height<<" X "<<width<<endl;
        cout<<"run "<<iterations<<" steps of life of game"<<endl;
        if(periodical)
            cout<<"with periodical boundary condition"<<endl;
        else
            cout<<"with non periodical boundary condition"<<endl;
    }
    //divide p processors into a 2D grid
	find_dimensions(p, id, rows, columns);
	if (id==0)
	{
		//print size to the file
		fstream size;
		string a = "size.txt";
		size.open(a, fstream::out);
		if (size.fail())
		{
			cout << "Error opening file" << endl;
			exit(0);
		}
		size << rows << " " << columns <<" "<<iterations<<" "<<int(periodical)<< endl;
		size.close();
	}

    //get the corresponding index in the grid
	id_to_index(id, id_row, id_column);


	//decide the gridsize for each processor
	//This section distributes the regions that processors are responsible for
    //code from solution 5_1
	int rows_responsible_min_max[2];
	int cols_responsible_min_max[2];
	vector< vector<int> > rows_responsible_list(rows, vector<int>(2));
	vector< vector<int> > cols_responsible_list(columns, vector<int>(2));

#ifdef DO_TIMING
	//The timing starts here 
    // exclude the initialisation of MPI, srand and the memory allocation
	auto start = chrono::high_resolution_clock::now();
#endif

	if (id == 0)
	{
		int cnt = 0;
		MPI_Request* decom_requests = new MPI_Request[(p - 1)*2];
		//domain decomposition by row of p processors
		decomposition(height, rows, rows_responsible_list);
		for (int j = 0; j < columns; j++)
		{
			for (int i = 0; i < rows; i++)
			{
				int real_id = id_from_index(i, j);
				if (real_id == 0)
				{
					for (int k = 0; k < 2; k++) rows_responsible_min_max[k] = rows_responsible_list[0][k];
				}
				else
				{
					MPI_Isend(&rows_responsible_list[i][0], 2, MPI_INT, real_id, 1, MPI_COMM_WORLD, &decom_requests[cnt++]);
				}

			}
		}
		//domain decomposition by columns of p processors
		decomposition(width, columns, cols_responsible_list);
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < columns; j++)
			{
				int real_id = id_from_index(i, j);
				if (real_id == 0)
				{
					for (int k = 0; k < 2; k++) cols_responsible_min_max[k] = cols_responsible_list[0][k];
				}
				else
				{
					MPI_Isend(&cols_responsible_list[j][0], 2, MPI_INT, real_id, 2, MPI_COMM_WORLD, &decom_requests[cnt++]);
				}

			}
		}
			

		MPI_Waitall(cnt, decom_requests, MPI_STATUSES_IGNORE);

		delete[] decom_requests;
	}
	else
	{
		//Non-blocking doesn't help here
		MPI_Recv(rows_responsible_min_max, 2, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(cols_responsible_min_max, 2, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	

	//intialise matrix
	set_grid(grid_rows, rows_responsible_min_max);
	set_grid(grid_cols, cols_responsible_min_max);
	bool* current = new bool[grid_rows * grid_cols]();
	bool* grid_ext = new bool[(grid_rows + 2) * (grid_cols + 2)]();
	//randomly set live_cells percentage of cells in the grid to be alive
	initialize(current, live_cells);
	copygrid(current, grid_ext);

#ifndef DO_TIMING
    //exclude couts when timing the code
	cout << id << " : rows " << grid_rows << ", cols: " <<grid_cols << endl;
#endif

	//create new mpi type of the column of matrix
	MPI_Type_vector(grid_rows, 1, grid_cols, MPI_C_BOOL, &curr_col);
	MPI_Type_commit(&curr_col);

	MPI_Type_vector(grid_rows, 1, grid_cols+2, MPI_C_BOOL, &ext_col);
	MPI_Type_commit(&ext_col);

#ifndef DO_TIMING	
    //exclude file writes when timing the code
	//print matrix to the file
	fstream matrixA;
	string a = "0_" + to_string(id) + ".txt";
	matrixA.open(a, fstream::out);
	if (matrixA.fail())
	{
		cout << "Error opening file" << endl;
		exit(0);
	}
	for (int i = 0; i < grid_rows; i++) {
		for (int j = 0; j < grid_cols; j++) {
			matrixA << current[i * grid_cols+j] << " ";
		}
		matrixA << endl;
	}
	matrixA.close();
#endif
	
	MPI_Request* request;
	
	//start life of game
	int steps = 0;
	while (iterations>0)
	{
		//send neighbors cells to grids
        //receive neighbors in the grid_ext
		request = new MPI_Request[16];
		int cnt = 0;
        //for the periodical boudary condition
		if (periodical)
		{
			for (int i = -1; i <= 1; i++)
				for (int j = -1; j <= 1; j++)
				{
					int com_i = id_row + i;
					int com_j = id_column + j;
					if (com_i >= rows)
						com_i %= rows;
					if (com_i < 0)
						com_i += rows;
					if (com_j >= columns)
						com_j %= columns;
					if (com_j < 0)
						com_j += columns;

					int com_id = id_from_index(com_i, com_j);
					
					if (!(i == 0 && j == 0))
					{
						send_neighbor( i, j, com_id, current, request, cnt);
						update_neighbor( i, j, com_id, grid_ext, request, cnt);
					}
				}
		}
        // for fixed boundaries
		else 
		{
			for (int i = -1; i <= 1; i++)
				for (int j = -1; j <= 1; j++)
				{
					int com_i = id_row + i;
					int com_j = id_column + j;

					int com_id = id_from_index(com_i, com_j);

					if (com_id != id && com_id >= 0 && com_id < p)
					{

						send_neighbor( i, j, com_id, current, request, cnt);
						update_neighbor( i, j, com_id, grid_ext, request, cnt);
					}
				}
		}
	
		MPI_Waitall(cnt, request, MPI_STATUS_IGNORE);
		delete[] request;

		//determine the life state of next step
		determineState(current, grid_ext);
		copygrid(current, grid_ext);
		iterations--;
		steps++;
    #ifndef DO_TIMING	
    //exclude file writes when timing the code
		a = to_string(steps) + "_" + to_string(id) + ".txt";
		matrixA.open(a, fstream::out);
		if (matrixA.fail())
		{
			cout << "Error opening file" << endl;
			exit(0);
		}
		for (int i = 0; i < grid_rows; i++) {
			for (int j = 0; j < grid_cols; j++) {
				matrixA << current[i * (grid_cols)+j] << " ";
			}
			matrixA << endl;
		}
		matrixA.close();
    #endif
	}

    //make sure every processor finished
	MPI_Barrier(MPI_COMM_WORLD);

#ifdef DO_TIMING
	auto finish = chrono::high_resolution_clock::now();
	if (id == 0)
	{
		std::chrono::duration<double> elapsed = finish - start;
		cout << setprecision(5);
		cout << "The code took " << elapsed.count() << "s to run" << endl;
	}
#endif

	delete[] current;
	delete[] grid_ext;
    MPI_Finalize();
}

// divide P processors into a grid size of rows*columns, code from ex3 of worksheet2
void find_dimensions(int p, int id, int& rows, int& columns)		
{
	int min_gap = p;
	int top = sqrt(p) + 1;
	for (int i = 1; i <= top; i++)
	{
		if (p % i == 0)
		{
			int gap = abs(p / i - i);

			if (gap < min_gap)
			{
				min_gap = gap;
				rows = i;
				columns = p / i;
			}
		}
	}

	if (id == 0)
		std::cout << "Divide " << p << " processors into " << rows << " by " << columns << " grid" << std::endl;
}
// find the index in the grid according to its id, code from ex3 of worksheet2
void id_to_index(int id, int& id_row, int& id_column)
{
	id_column = id % columns;
	id_row = id / columns;
}
// find its id according to its index in the grid, code from ex3 of worksheet2
int id_from_index(int id_row, int id_column)
{
	if (id_row >= rows || id_row < 0)
		return -1;
	if (id_column >= columns || id_column < 0)
		return -1;

	return id_row * columns + id_column;
}

// set the grid size for each processor
void set_grid(int& grid_rows, int rows_responsible_min_max[2]) {
	grid_rows = rows_responsible_min_max[1] - rows_responsible_min_max[0] + 1;
}
// randomly set per percentage of alive cells in each grid
void initialize(bool *current, double per) {
	
	int length = grid_cols * grid_rows;
	int true_number = int(length * per);
	if (true_number < 1)
	{
		std::cout << "Warning: no true element in return array" << std::endl;
		return;
	}
	int* index = new int[length];
	for (int i = 0; i < length; ++i) index[i] = i;
	for (int i = length - 1; i >= 1; --i) std::swap(index[i], index[rand() % i]);
	for (int i = 0; i < true_number; i++) current[index[i]] = true;
	
	delete[] index;
	
}

void copygrid(bool* current, bool* grid_ext)
{
	for (int i = 1; i <= grid_rows; i++)
	{
		for (int j = 1; j <= grid_cols; j++)
		{
			grid_ext[i * (grid_cols + 2) + j] = current[(i - 1) * grid_cols + j - 1];
		}
	}

}
// determine the next life state inside grids
// Each cell has eight neighbours (vertical, horizontal and diagonal)
// If a living cell has fewer than 2 neighbours it dies (not enough to breed) 
// If a living cell has 2 or 3 neighbours it survives
// If a living cell has 4 or more neighbours it dies (over population)
// If a dead cell has exactly 3 neighbours a living cell is born there
void determineState(bool* current, bool* grid_ext) {
	for (int i = 1; i <= grid_rows; i++)
	{
		for (int j = 1; j <= grid_cols; j++)
		{
			int alive = 0;
			for (int k = -1; k <= 1; k++)
			{
				for (int m = -1; m <= 1; m++)
				{
					if (!(k == 0 && m == 0))
					{
						if (grid_ext[(i + k) * (grid_cols + 2) + j + m])
						{
							++alive;
						}
					}
				}
			}
			if (alive < 2)
			{
				current[(i - 1) * grid_cols + j - 1] = false;
			}
			else if (alive == 3)
			{
				current[(i - 1) * grid_cols + j - 1] = true;
			}
			else if (alive > 3)
			{
				current[(i - 1) * grid_cols + j - 1] = false;
			}
		}
	}


}
// calculate the grid size for each processor
void decomposition(int height, int rows, vector< vector<int> > &rows_responsible_list) {
	int rows_left = height;
	int min_row = 0;
	for (int i = 0; i < rows; i++)
	{
		//This is done in case the number of rows are not exactly divisible by p
		//Better than having all the shortfall accumulate on one processor
		int portion = rows_left / (rows- i);
		rows_responsible_list[i][0] = min_row;
		rows_responsible_list[i][1] = min_row + portion - 1;
		min_row += portion;
		rows_left -= portion;
	}

}

void send_neighbor( int i, int j, int com_id, bool* current, MPI_Request* request, int &cnt)
{
	//top-left
	if (i == -1 && j == -1) {
		MPI_Isend(&current[0], 1, MPI_C_BOOL, com_id, tag_num, MPI_COMM_WORLD, &request[cnt++]);
	}
	//top
	if (i == -1 && j == 0) {
		MPI_Isend(&current[0], grid_cols, MPI_C_BOOL, com_id, tag_num+1, MPI_COMM_WORLD, &request[cnt++]);
	}
	//top-right
	if (i == -1 && j == 1) {
		MPI_Isend(&current[grid_cols - 1], 1, MPI_C_BOOL, com_id, tag_num+2, MPI_COMM_WORLD, &request[cnt++]);
	}
	//left
	if (i == 0 && j == -1) {
		MPI_Isend(&current[0], 1, curr_col, com_id, tag_num+3, MPI_COMM_WORLD, &request[cnt++]);
	}
	//right
	if (i == 0 && j == 1) {
		MPI_Isend(&current[grid_cols - 1], 1, curr_col, com_id, tag_num+4, MPI_COMM_WORLD, &request[cnt++]);
	}
	
	//bottom-left
	if (i == 1 && j == -1) {
		MPI_Isend(&current[(grid_rows - 1) * grid_cols], 1, MPI_C_BOOL, com_id, tag_num+5, MPI_COMM_WORLD, &request[cnt++]);
	}
	//bottom
	if (i == 1 && j == 0) {
		MPI_Isend(&current[(grid_rows - 1) * grid_cols], grid_cols, MPI_C_BOOL, com_id, tag_num+6, MPI_COMM_WORLD, &request[cnt++]);
	}
	//bottom-right
	if (i == 1 && j == 1) {
		MPI_Isend(&current[(grid_rows - 1) * grid_cols + grid_cols - 1], 1, MPI_C_BOOL, com_id, tag_num+7, MPI_COMM_WORLD, &request[cnt++]);
	}
}
void update_neighbor(int i, int j, int com_id, bool* grid_ext, MPI_Request* request, int &cnt)
{
	//top-left
	if (i == -1 && j == -1) {
		MPI_Irecv(&grid_ext[0], 1, MPI_C_BOOL, com_id, tag_num+7, MPI_COMM_WORLD, &request[cnt++]);
	}
	//top
	if (i == -1 && j == 0) {
		MPI_Irecv(&grid_ext[1], grid_cols, MPI_C_BOOL, com_id, tag_num+6, MPI_COMM_WORLD, &request[cnt++]);
	}
	//top-right
	if (i == -1 && j == 1) {

		MPI_Irecv(&grid_ext[grid_cols + 1], 1, MPI_C_BOOL, com_id, tag_num+5, MPI_COMM_WORLD, &request[cnt++]);
	}
	//right
	if (i == 0 && j == 1) {

		MPI_Irecv(&grid_ext[grid_cols + 2 + grid_cols + 1], 1, ext_col, com_id, tag_num+3, MPI_COMM_WORLD, &request[cnt++]);
	}
	//left
	if (i == 0 && j == -1) {
		
		MPI_Irecv(&grid_ext[grid_cols + 2], 1, ext_col, com_id, tag_num+4, MPI_COMM_WORLD, &request[cnt++]);
	}
	//bottom-left
	if (i == 1 && j == -1) {
		MPI_Irecv(&grid_ext[(grid_rows + 1) * (grid_cols + 2)], 1, MPI_C_BOOL, com_id, tag_num+2, MPI_COMM_WORLD, &request[cnt++]);
	}
	//bottom
	if (i == 1 && j == 0) {
		MPI_Irecv(&grid_ext[(grid_rows + 1) * (grid_cols + 2) + 1], grid_cols, MPI_C_BOOL, com_id, tag_num+1, MPI_COMM_WORLD, &request[cnt++]);
	}
	//bottom-right
	if (i == 1 && j == 1) {
		MPI_Irecv(&grid_ext[(grid_rows + 1) * (grid_cols + 2) + grid_cols + 1], 1, MPI_C_BOOL, com_id, tag_num, MPI_COMM_WORLD, &request[cnt++]);
	}
}
