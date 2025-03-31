; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 234553

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child431 child107 child281 - child
    bread122 bread167 bread253 - bread-portion
    content271 content294 content401 - content-portion
    tray416 tray361 tray224 tray481 - tray
    table199 table320 table124 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray416 kitchen)
     (at tray361 kitchen)
     (at tray224 kitchen)
     (at tray481 kitchen)
     (at_kitchen_bread bread122)
     (at_kitchen_bread bread167)
     (at_kitchen_bread bread253)
     (at_kitchen_content content271)
     (at_kitchen_content content294)
     (at_kitchen_content content401)
     (not_allergic_gluten child431)
     (not_allergic_gluten child281)
     (not_allergic_gluten child107)
     (waiting child431 table124)
     (waiting child107 table199)
     (waiting child281 table199)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child431)
     (served child107)
     (served child281)
    )
  )
)
