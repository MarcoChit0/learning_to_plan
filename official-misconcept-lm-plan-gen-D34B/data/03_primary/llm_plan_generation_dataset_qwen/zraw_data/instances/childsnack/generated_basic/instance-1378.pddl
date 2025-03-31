; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 826847

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child171 child107 - child
    bread167 bread253 - bread-portion
    content133 content150 - content-portion
    tray134 tray204 tray301 - tray
    table204 table304 table351 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray134 kitchen)
     (at tray204 kitchen)
     (at tray301 kitchen)
     (at_kitchen_bread bread167)
     (at_kitchen_bread bread253)
     (at_kitchen_content content133)
     (at_kitchen_content content150)
     (not_allergic_gluten child107)
     (not_allergic_gluten child171)
     (waiting child171 table351)
     (waiting child107 table204)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child171)
     (served child107)
    )
  )
)
