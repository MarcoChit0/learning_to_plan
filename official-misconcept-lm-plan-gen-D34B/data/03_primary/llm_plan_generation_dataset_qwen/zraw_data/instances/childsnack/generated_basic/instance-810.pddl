; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 912470

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child158 - child
    bread388 - bread-portion
    content345 - content-portion
    tray309 tray149 - tray
    table300 table154 table181 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray309 kitchen)
     (at tray149 kitchen)
     (at_kitchen_bread bread388)
     (at_kitchen_content content345)
     (not_allergic_gluten child158)
     (waiting child158 table181)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child158)
    )
  )
)
