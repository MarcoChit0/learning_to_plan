; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 864363

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child311 child114 - child
    bread384 bread26 - bread-portion
    content181 content152 - content-portion
    tray330 tray33 tray405 - tray
    table53 table221 table343 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray330 kitchen)
     (at tray33 kitchen)
     (at tray405 kitchen)
     (at_kitchen_bread bread384)
     (at_kitchen_bread bread26)
     (at_kitchen_content content181)
     (at_kitchen_content content152)
     (not_allergic_gluten child311)
     (not_allergic_gluten child114)
     (waiting child311 table221)
     (waiting child114 table221)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child311)
     (served child114)
    )
  )
)
