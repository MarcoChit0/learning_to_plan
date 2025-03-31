; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 420973

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child177 - child
    bread433 - bread-portion
    content435 - content-portion
    tray114 tray115 tray237 tray9 - tray
    table439 table63 table447 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray114 kitchen)
     (at tray115 kitchen)
     (at tray237 kitchen)
     (at tray9 kitchen)
     (at_kitchen_bread bread433)
     (at_kitchen_content content435)
     (not_allergic_gluten child177)
     (waiting child177 table63)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child177)
    )
  )
)
