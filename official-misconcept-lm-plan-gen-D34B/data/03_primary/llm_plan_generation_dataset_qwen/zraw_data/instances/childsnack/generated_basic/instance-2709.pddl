; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 968221

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child439 child482 child89 - child
    bread406 bread52 bread323 - bread-portion
    content382 content121 content168 - content-portion
    tray345 tray220 tray467 - tray
    table219 table372 table337 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray345 kitchen)
     (at tray220 kitchen)
     (at tray467 kitchen)
     (at_kitchen_bread bread406)
     (at_kitchen_bread bread52)
     (at_kitchen_bread bread323)
     (at_kitchen_content content382)
     (at_kitchen_content content121)
     (at_kitchen_content content168)
     (not_allergic_gluten child89)
     (not_allergic_gluten child482)
     (not_allergic_gluten child439)
     (waiting child439 table219)
     (waiting child482 table219)
     (waiting child89 table219)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child439)
     (served child482)
     (served child89)
    )
  )
)
