; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 814427

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child488 child407 - child
    bread278 bread354 - bread-portion
    content177 content451 - content-portion
    tray131 - tray
    table46 table235 table236 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray131 kitchen)
     (at_kitchen_bread bread278)
     (at_kitchen_bread bread354)
     (at_kitchen_content content177)
     (at_kitchen_content content451)
     (no_gluten_bread bread354)
     (no_gluten_content content177)
     (allergic_gluten child488)
     (not_allergic_gluten child407)
     (waiting child488 table236)
     (waiting child407 table46)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child488)
     (served child407)
    )
  )
)
