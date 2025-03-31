; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 702715

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child190 - child
    bread479 - bread-portion
    content53 - content-portion
    tray481 - tray
    table40 table85 table338 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray481 kitchen)
     (at_kitchen_bread bread479)
     (at_kitchen_content content53)
     (not_allergic_gluten child190)
     (waiting child190 table338)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child190)
    )
  )
)
