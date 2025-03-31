; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 424098

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child40 - child
    bread422 - bread-portion
    content434 - content-portion
    tray248 tray322 tray26 - tray
    table480 table356 table408 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray248 kitchen)
     (at tray322 kitchen)
     (at tray26 kitchen)
     (at_kitchen_bread bread422)
     (at_kitchen_content content434)
     (not_allergic_gluten child40)
     (waiting child40 table480)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child40)
    )
  )
)
