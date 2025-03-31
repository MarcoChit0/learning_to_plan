; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 382520

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child254 - child
    bread268 - bread-portion
    content429 - content-portion
    tray471 tray49 tray433 tray32 - tray
    table106 table119 table298 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray471 kitchen)
     (at tray49 kitchen)
     (at tray433 kitchen)
     (at tray32 kitchen)
     (at_kitchen_bread bread268)
     (at_kitchen_content content429)
     (not_allergic_gluten child254)
     (waiting child254 table298)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child254)
    )
  )
)
