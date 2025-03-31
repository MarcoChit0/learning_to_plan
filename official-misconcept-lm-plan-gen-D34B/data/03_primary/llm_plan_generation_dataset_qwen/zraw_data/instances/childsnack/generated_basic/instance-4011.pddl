; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 578615

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child189 child44 child306 - child
    bread378 bread96 bread205 - bread-portion
    content23 content486 content219 - content-portion
    tray56 - tray
    table39 table116 table401 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray56 kitchen)
     (at_kitchen_bread bread378)
     (at_kitchen_bread bread96)
     (at_kitchen_bread bread205)
     (at_kitchen_content content23)
     (at_kitchen_content content486)
     (at_kitchen_content content219)
     (not_allergic_gluten child189)
     (not_allergic_gluten child44)
     (not_allergic_gluten child306)
     (waiting child189 table401)
     (waiting child44 table116)
     (waiting child306 table116)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child189)
     (served child44)
     (served child306)
    )
  )
)
