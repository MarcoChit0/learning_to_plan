; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 743430

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child195 - child
    bread34 - bread-portion
    content310 - content-portion
    tray58 tray250 tray311 tray8 - tray
    table273 table83 table441 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray58 kitchen)
     (at tray250 kitchen)
     (at tray311 kitchen)
     (at tray8 kitchen)
     (at_kitchen_bread bread34)
     (at_kitchen_content content310)
     (not_allergic_gluten child195)
     (waiting child195 table273)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child195)
    )
  )
)
