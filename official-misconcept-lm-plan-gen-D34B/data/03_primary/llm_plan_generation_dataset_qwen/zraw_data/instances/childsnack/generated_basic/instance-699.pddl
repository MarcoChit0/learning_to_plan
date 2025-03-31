; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 217473

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child419 - child
    bread430 - bread-portion
    content351 - content-portion
    tray53 tray93 tray52 tray453 - tray
    table241 table466 table288 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray53 kitchen)
     (at tray93 kitchen)
     (at tray52 kitchen)
     (at tray453 kitchen)
     (at_kitchen_bread bread430)
     (at_kitchen_content content351)
     (not_allergic_gluten child419)
     (waiting child419 table288)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child419)
    )
  )
)
