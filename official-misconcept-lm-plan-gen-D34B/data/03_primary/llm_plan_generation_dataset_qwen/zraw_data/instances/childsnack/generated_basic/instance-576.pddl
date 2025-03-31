; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 75650

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child198 - child
    bread358 - bread-portion
    content27 - content-portion
    tray491 - tray
    table90 table6 table387 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray491 kitchen)
     (at_kitchen_bread bread358)
     (at_kitchen_content content27)
     (not_allergic_gluten child198)
     (waiting child198 table387)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child198)
    )
  )
)
