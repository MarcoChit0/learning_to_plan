; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 237273

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child336 - child
    bread47 - bread-portion
    content255 - content-portion
    tray326 tray498 tray208 tray73 - tray
    table360 table333 table8 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray326 kitchen)
     (at tray498 kitchen)
     (at tray208 kitchen)
     (at tray73 kitchen)
     (at_kitchen_bread bread47)
     (at_kitchen_content content255)
     (not_allergic_gluten child336)
     (waiting child336 table8)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child336)
    )
  )
)
