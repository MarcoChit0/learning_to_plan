; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 309172

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child405 child3 - child
    bread237 bread135 - bread-portion
    content102 content298 - content-portion
    tray245 tray321 tray0 - tray
    table212 table272 table407 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray245 kitchen)
     (at tray321 kitchen)
     (at tray0 kitchen)
     (at_kitchen_bread bread237)
     (at_kitchen_bread bread135)
     (at_kitchen_content content102)
     (at_kitchen_content content298)
     (not_allergic_gluten child3)
     (not_allergic_gluten child405)
     (waiting child405 table272)
     (waiting child3 table272)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child405)
     (served child3)
    )
  )
)
