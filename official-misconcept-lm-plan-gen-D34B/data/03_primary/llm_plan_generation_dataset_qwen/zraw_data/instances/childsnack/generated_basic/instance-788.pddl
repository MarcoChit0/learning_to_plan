; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 479166

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child170 - child
    bread8 - bread-portion
    content144 - content-portion
    tray499 tray4 - tray
    table299 table416 table444 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray499 kitchen)
     (at tray4 kitchen)
     (at_kitchen_bread bread8)
     (at_kitchen_content content144)
     (not_allergic_gluten child170)
     (waiting child170 table416)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child170)
    )
  )
)
