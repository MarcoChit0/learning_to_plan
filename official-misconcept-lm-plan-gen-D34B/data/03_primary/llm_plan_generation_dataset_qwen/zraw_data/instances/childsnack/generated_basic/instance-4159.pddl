; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 7353

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child45 child252 - child
    bread442 bread413 - bread-portion
    content426 content480 - content-portion
    tray399 tray6 tray10 - tray
    table193 table144 table368 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray399 kitchen)
     (at tray6 kitchen)
     (at tray10 kitchen)
     (at_kitchen_bread bread442)
     (at_kitchen_bread bread413)
     (at_kitchen_content content426)
     (at_kitchen_content content480)
     (not_allergic_gluten child252)
     (not_allergic_gluten child45)
     (waiting child45 table144)
     (waiting child252 table193)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child45)
     (served child252)
    )
  )
)
