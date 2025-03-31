; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 557880

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child133 - child
    bread122 - bread-portion
    content15 - content-portion
    tray315 tray476 tray404 tray84 - tray
    table403 table473 table45 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray315 kitchen)
     (at tray476 kitchen)
     (at tray404 kitchen)
     (at tray84 kitchen)
     (at_kitchen_bread bread122)
     (at_kitchen_content content15)
     (not_allergic_gluten child133)
     (waiting child133 table473)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child133)
    )
  )
)
