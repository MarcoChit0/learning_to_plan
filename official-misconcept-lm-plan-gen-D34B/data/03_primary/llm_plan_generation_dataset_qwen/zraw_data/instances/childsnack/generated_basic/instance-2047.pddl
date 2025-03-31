; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 600275

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child406 child56 - child
    bread321 bread471 - bread-portion
    content285 content19 - content-portion
    tray121 tray380 tray257 - tray
    table366 table420 table452 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray121 kitchen)
     (at tray380 kitchen)
     (at tray257 kitchen)
     (at_kitchen_bread bread321)
     (at_kitchen_bread bread471)
     (at_kitchen_content content285)
     (at_kitchen_content content19)
     (not_allergic_gluten child406)
     (not_allergic_gluten child56)
     (waiting child406 table420)
     (waiting child56 table452)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child406)
     (served child56)
    )
  )
)
