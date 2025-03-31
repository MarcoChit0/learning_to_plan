; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 619541

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child289 child350 - child
    bread407 bread493 - bread-portion
    content253 content260 - content-portion
    tray320 tray366 - tray
    table440 table258 table454 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray320 kitchen)
     (at tray366 kitchen)
     (at_kitchen_bread bread407)
     (at_kitchen_bread bread493)
     (at_kitchen_content content253)
     (at_kitchen_content content260)
     (no_gluten_bread bread493)
     (no_gluten_content content260)
     (allergic_gluten child350)
     (not_allergic_gluten child289)
     (waiting child289 table258)
     (waiting child350 table454)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child289)
     (served child350)
    )
  )
)
