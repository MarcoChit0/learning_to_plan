; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 748910

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child33 - child
    bread430 - bread-portion
    content4 - content-portion
    tray200 tray142 tray324 - tray
    table334 table399 table267 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray200 kitchen)
     (at tray142 kitchen)
     (at tray324 kitchen)
     (at_kitchen_bread bread430)
     (at_kitchen_content content4)
     (not_allergic_gluten child33)
     (waiting child33 table267)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child33)
    )
  )
)
