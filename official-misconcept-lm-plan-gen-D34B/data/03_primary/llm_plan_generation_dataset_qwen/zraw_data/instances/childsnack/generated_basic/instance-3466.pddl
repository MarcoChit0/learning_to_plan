; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 987208

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child239 child329 - child
    bread376 bread404 - bread-portion
    content327 content285 - content-portion
    tray419 tray460 - tray
    table440 table286 table73 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray419 kitchen)
     (at tray460 kitchen)
     (at_kitchen_bread bread376)
     (at_kitchen_bread bread404)
     (at_kitchen_content content327)
     (at_kitchen_content content285)
     (no_gluten_bread bread404)
     (no_gluten_content content327)
     (allergic_gluten child239)
     (not_allergic_gluten child329)
     (waiting child239 table286)
     (waiting child329 table440)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child239)
     (served child329)
    )
  )
)
