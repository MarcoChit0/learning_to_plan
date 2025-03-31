; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 915769

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child142 child80 - child
    bread388 bread78 - bread-portion
    content382 content168 - content-portion
    tray21 tray434 tray267 - tray
    table379 table82 table312 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray21 kitchen)
     (at tray434 kitchen)
     (at tray267 kitchen)
     (at_kitchen_bread bread388)
     (at_kitchen_bread bread78)
     (at_kitchen_content content382)
     (at_kitchen_content content168)
     (not_allergic_gluten child80)
     (not_allergic_gluten child142)
     (waiting child142 table82)
     (waiting child80 table379)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child142)
     (served child80)
    )
  )
)
