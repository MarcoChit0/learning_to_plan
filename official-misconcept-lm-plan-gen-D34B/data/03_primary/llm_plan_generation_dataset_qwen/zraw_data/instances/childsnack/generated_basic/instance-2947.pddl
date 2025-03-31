; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 327039

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child317 - child
    bread447 - bread-portion
    content200 - content-portion
    tray445 tray105 tray357 tray493 - tray
    table135 table272 table361 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray445 kitchen)
     (at tray105 kitchen)
     (at tray357 kitchen)
     (at tray493 kitchen)
     (at_kitchen_bread bread447)
     (at_kitchen_content content200)
     (not_allergic_gluten child317)
     (waiting child317 table135)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child317)
    )
  )
)
