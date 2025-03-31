; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 25340

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child421 child42 - child
    bread158 bread257 - bread-portion
    content323 content414 - content-portion
    tray450 tray171 - tray
    table63 table427 table351 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray450 kitchen)
     (at tray171 kitchen)
     (at_kitchen_bread bread158)
     (at_kitchen_bread bread257)
     (at_kitchen_content content323)
     (at_kitchen_content content414)
     (not_allergic_gluten child42)
     (not_allergic_gluten child421)
     (waiting child421 table351)
     (waiting child42 table427)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child421)
     (served child42)
    )
  )
)
