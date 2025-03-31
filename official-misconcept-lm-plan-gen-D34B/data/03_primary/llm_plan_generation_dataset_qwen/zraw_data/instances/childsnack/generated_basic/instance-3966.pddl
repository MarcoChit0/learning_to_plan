; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 506019

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child195 - child
    bread224 - bread-portion
    content15 - content-portion
    tray52 tray27 tray117 - tray
    table471 table174 table157 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray52 kitchen)
     (at tray27 kitchen)
     (at tray117 kitchen)
     (at_kitchen_bread bread224)
     (at_kitchen_content content15)
     (not_allergic_gluten child195)
     (waiting child195 table174)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child195)
    )
  )
)
