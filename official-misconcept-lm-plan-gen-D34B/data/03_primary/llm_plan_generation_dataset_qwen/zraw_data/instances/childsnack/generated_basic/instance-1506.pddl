; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 726582

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child230 child497 - child
    bread447 bread91 - bread-portion
    content172 content94 - content-portion
    tray201 tray3 tray240 - tray
    table75 table303 table102 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray201 kitchen)
     (at tray3 kitchen)
     (at tray240 kitchen)
     (at_kitchen_bread bread447)
     (at_kitchen_bread bread91)
     (at_kitchen_content content172)
     (at_kitchen_content content94)
     (no_gluten_bread bread91)
     (no_gluten_content content94)
     (allergic_gluten child497)
     (not_allergic_gluten child230)
     (waiting child230 table75)
     (waiting child497 table303)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child230)
     (served child497)
    )
  )
)
