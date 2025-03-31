; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 680745

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child464 child269 - child
    bread197 bread50 - bread-portion
    content201 content73 - content-portion
    tray335 tray389 - tray
    table497 table103 table490 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray335 kitchen)
     (at tray389 kitchen)
     (at_kitchen_bread bread197)
     (at_kitchen_bread bread50)
     (at_kitchen_content content201)
     (at_kitchen_content content73)
     (not_allergic_gluten child464)
     (not_allergic_gluten child269)
     (waiting child464 table497)
     (waiting child269 table490)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child464)
     (served child269)
    )
  )
)
