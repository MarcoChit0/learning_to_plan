; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 35455

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child364 child342 - child
    bread124 bread380 - bread-portion
    content443 content228 - content-portion
    tray20 tray334 - tray
    table439 table437 table360 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray20 kitchen)
     (at tray334 kitchen)
     (at_kitchen_bread bread124)
     (at_kitchen_bread bread380)
     (at_kitchen_content content443)
     (at_kitchen_content content228)
     (no_gluten_bread bread380)
     (no_gluten_content content443)
     (allergic_gluten child364)
     (not_allergic_gluten child342)
     (waiting child364 table437)
     (waiting child342 table360)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child364)
     (served child342)
    )
  )
)
