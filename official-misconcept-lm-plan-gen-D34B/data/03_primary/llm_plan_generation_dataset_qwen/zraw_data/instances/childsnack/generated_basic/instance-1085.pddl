; child-snack task with 3 children and 0.38 gluten factor 
; constant factor of 1.5
; random seed: 48357

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child261 child470 child352 - child
    bread9 bread10 bread91 - bread-portion
    content24 content327 content474 - content-portion
    tray305 tray321 tray220 - tray
    table200 table316 table232 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray305 kitchen)
     (at tray321 kitchen)
     (at tray220 kitchen)
     (at_kitchen_bread bread9)
     (at_kitchen_bread bread10)
     (at_kitchen_bread bread91)
     (at_kitchen_content content24)
     (at_kitchen_content content327)
     (at_kitchen_content content474)
     (no_gluten_bread bread91)
     (no_gluten_content content474)
     (allergic_gluten child261)
     (not_allergic_gluten child470)
     (not_allergic_gluten child352)
     (waiting child261 table200)
     (waiting child470 table316)
     (waiting child352 table316)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child261)
     (served child470)
     (served child352)
    )
  )
)
