; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 505902

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child282 child350 - child
    bread404 bread487 - bread-portion
    content163 content191 - content-portion
    tray113 tray283 tray56 - tray
    table97 table93 table270 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray113 kitchen)
     (at tray283 kitchen)
     (at tray56 kitchen)
     (at_kitchen_bread bread404)
     (at_kitchen_bread bread487)
     (at_kitchen_content content163)
     (at_kitchen_content content191)
     (not_allergic_gluten child282)
     (not_allergic_gluten child350)
     (waiting child282 table270)
     (waiting child350 table270)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child282)
     (served child350)
    )
  )
)
