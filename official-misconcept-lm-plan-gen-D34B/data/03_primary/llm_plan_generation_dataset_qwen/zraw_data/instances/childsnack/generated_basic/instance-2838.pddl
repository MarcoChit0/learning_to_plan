; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 927229

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child94 child231 - child
    bread436 bread375 - bread-portion
    content496 content118 - content-portion
    tray427 tray142 - tray
    table78 table432 table217 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray427 kitchen)
     (at tray142 kitchen)
     (at_kitchen_bread bread436)
     (at_kitchen_bread bread375)
     (at_kitchen_content content496)
     (at_kitchen_content content118)
     (not_allergic_gluten child231)
     (not_allergic_gluten child94)
     (waiting child94 table78)
     (waiting child231 table217)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child94)
     (served child231)
    )
  )
)
