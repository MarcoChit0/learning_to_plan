; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 73302

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child407 child166 - child
    bread255 bread37 - bread-portion
    content426 content336 - content-portion
    tray372 tray410 tray17 - tray
    table342 table173 table423 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray372 kitchen)
     (at tray410 kitchen)
     (at tray17 kitchen)
     (at_kitchen_bread bread255)
     (at_kitchen_bread bread37)
     (at_kitchen_content content426)
     (at_kitchen_content content336)
     (not_allergic_gluten child166)
     (not_allergic_gluten child407)
     (waiting child407 table423)
     (waiting child166 table342)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child407)
     (served child166)
    )
  )
)
