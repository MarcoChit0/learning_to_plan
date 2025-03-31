; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 364412

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child391 child442 - child
    bread359 bread299 - bread-portion
    content395 content319 - content-portion
    tray47 tray288 - tray
    table181 table308 table468 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray47 kitchen)
     (at tray288 kitchen)
     (at_kitchen_bread bread359)
     (at_kitchen_bread bread299)
     (at_kitchen_content content395)
     (at_kitchen_content content319)
     (not_allergic_gluten child442)
     (not_allergic_gluten child391)
     (waiting child391 table181)
     (waiting child442 table308)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child391)
     (served child442)
    )
  )
)
