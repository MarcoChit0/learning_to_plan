; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 559369

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child326 - child
    bread472 - bread-portion
    content471 - content-portion
    tray363 tray105 tray242 - tray
    table369 table56 table183 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray363 kitchen)
     (at tray105 kitchen)
     (at tray242 kitchen)
     (at_kitchen_bread bread472)
     (at_kitchen_content content471)
     (not_allergic_gluten child326)
     (waiting child326 table56)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child326)
    )
  )
)
