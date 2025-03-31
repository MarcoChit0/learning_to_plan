; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 26116

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child103 child463 - child
    bread106 bread143 - bread-portion
    content125 content270 - content-portion
    tray305 tray203 - tray
    table116 table288 table199 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray305 kitchen)
     (at tray203 kitchen)
     (at_kitchen_bread bread106)
     (at_kitchen_bread bread143)
     (at_kitchen_content content125)
     (at_kitchen_content content270)
     (not_allergic_gluten child103)
     (not_allergic_gluten child463)
     (waiting child103 table199)
     (waiting child463 table199)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child103)
     (served child463)
    )
  )
)
