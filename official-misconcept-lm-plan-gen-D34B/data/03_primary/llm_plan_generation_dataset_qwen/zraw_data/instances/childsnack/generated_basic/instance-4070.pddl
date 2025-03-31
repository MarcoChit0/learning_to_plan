; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 299022

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child256 - child
    bread106 - bread-portion
    content460 - content-portion
    tray498 tray372 tray287 - tray
    table53 table393 table99 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray498 kitchen)
     (at tray372 kitchen)
     (at tray287 kitchen)
     (at_kitchen_bread bread106)
     (at_kitchen_content content460)
     (not_allergic_gluten child256)
     (waiting child256 table53)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child256)
    )
  )
)
