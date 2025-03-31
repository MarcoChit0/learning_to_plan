; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 957882

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child7 child260 - child
    bread453 bread35 - bread-portion
    content144 content324 - content-portion
    tray169 tray439 tray150 - tray
    table414 table113 table419 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray169 kitchen)
     (at tray439 kitchen)
     (at tray150 kitchen)
     (at_kitchen_bread bread453)
     (at_kitchen_bread bread35)
     (at_kitchen_content content144)
     (at_kitchen_content content324)
     (no_gluten_bread bread35)
     (no_gluten_content content324)
     (allergic_gluten child7)
     (not_allergic_gluten child260)
     (waiting child7 table414)
     (waiting child260 table113)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child7)
     (served child260)
    )
  )
)
