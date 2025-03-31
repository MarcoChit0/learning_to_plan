; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 59885

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child417 - child
    bread84 - bread-portion
    content338 - content-portion
    tray443 tray342 tray61 tray239 - tray
    table147 table495 table135 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray443 kitchen)
     (at tray342 kitchen)
     (at tray61 kitchen)
     (at tray239 kitchen)
     (at_kitchen_bread bread84)
     (at_kitchen_content content338)
     (not_allergic_gluten child417)
     (waiting child417 table495)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child417)
    )
  )
)
