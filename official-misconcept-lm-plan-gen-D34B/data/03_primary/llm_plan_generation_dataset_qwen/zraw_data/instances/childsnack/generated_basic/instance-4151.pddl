; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 470607

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child414 child72 - child
    bread365 bread391 - bread-portion
    content121 content468 - content-portion
    tray257 tray203 tray422 - tray
    table428 table282 table314 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray257 kitchen)
     (at tray203 kitchen)
     (at tray422 kitchen)
     (at_kitchen_bread bread365)
     (at_kitchen_bread bread391)
     (at_kitchen_content content121)
     (at_kitchen_content content468)
     (not_allergic_gluten child414)
     (not_allergic_gluten child72)
     (waiting child414 table282)
     (waiting child72 table314)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child414)
     (served child72)
    )
  )
)
