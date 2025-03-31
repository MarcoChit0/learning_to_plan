; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 976727

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child309 child355 - child
    bread94 bread58 - bread-portion
    content53 content20 - content-portion
    tray173 tray410 - tray
    table13 table341 table111 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray173 kitchen)
     (at tray410 kitchen)
     (at_kitchen_bread bread94)
     (at_kitchen_bread bread58)
     (at_kitchen_content content53)
     (at_kitchen_content content20)
     (not_allergic_gluten child309)
     (not_allergic_gluten child355)
     (waiting child309 table341)
     (waiting child355 table111)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child309)
     (served child355)
    )
  )
)
