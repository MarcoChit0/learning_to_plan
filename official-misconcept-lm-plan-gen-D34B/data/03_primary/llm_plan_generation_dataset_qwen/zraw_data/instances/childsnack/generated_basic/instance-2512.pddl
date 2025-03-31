; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 745829

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child387 - child
    bread373 - bread-portion
    content353 - content-portion
    tray7 tray240 tray164 - tray
    table121 table275 table96 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray7 kitchen)
     (at tray240 kitchen)
     (at tray164 kitchen)
     (at_kitchen_bread bread373)
     (at_kitchen_content content353)
     (not_allergic_gluten child387)
     (waiting child387 table275)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child387)
    )
  )
)
