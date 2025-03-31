; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 241785

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child384 child197 - child
    bread355 bread24 - bread-portion
    content260 content465 - content-portion
    tray458 tray16 tray103 tray173 - tray
    table30 table272 table196 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray458 kitchen)
     (at tray16 kitchen)
     (at tray103 kitchen)
     (at tray173 kitchen)
     (at_kitchen_bread bread355)
     (at_kitchen_bread bread24)
     (at_kitchen_content content260)
     (at_kitchen_content content465)
     (not_allergic_gluten child197)
     (not_allergic_gluten child384)
     (waiting child384 table30)
     (waiting child197 table272)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child384)
     (served child197)
    )
  )
)
