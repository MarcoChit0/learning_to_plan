; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 652468

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child172 child400 - child
    bread50 bread48 - bread-portion
    content350 content230 - content-portion
    tray368 tray75 tray480 - tray
    table226 table175 table291 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray368 kitchen)
     (at tray75 kitchen)
     (at tray480 kitchen)
     (at_kitchen_bread bread50)
     (at_kitchen_bread bread48)
     (at_kitchen_content content350)
     (at_kitchen_content content230)
     (not_allergic_gluten child172)
     (not_allergic_gluten child400)
     (waiting child172 table291)
     (waiting child400 table175)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child172)
     (served child400)
    )
  )
)
