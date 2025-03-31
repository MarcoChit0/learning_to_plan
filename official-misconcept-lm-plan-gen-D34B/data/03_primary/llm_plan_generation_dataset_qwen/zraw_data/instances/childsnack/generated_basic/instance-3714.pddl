; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 923834

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child142 - child
    bread143 - bread-portion
    content121 - content-portion
    tray227 tray403 tray475 - tray
    table466 table141 table213 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray227 kitchen)
     (at tray403 kitchen)
     (at tray475 kitchen)
     (at_kitchen_bread bread143)
     (at_kitchen_content content121)
     (not_allergic_gluten child142)
     (waiting child142 table213)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child142)
    )
  )
)
