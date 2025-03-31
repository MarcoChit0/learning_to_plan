; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 393844

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child485 child494 - child
    bread440 bread118 - bread-portion
    content86 content48 - content-portion
    tray118 tray67 - tray
    table112 table184 table212 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray118 kitchen)
     (at tray67 kitchen)
     (at_kitchen_bread bread440)
     (at_kitchen_bread bread118)
     (at_kitchen_content content86)
     (at_kitchen_content content48)
     (not_allergic_gluten child494)
     (not_allergic_gluten child485)
     (waiting child485 table184)
     (waiting child494 table212)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child485)
     (served child494)
    )
  )
)
