; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 767014

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child107 - child
    bread423 - bread-portion
    content49 - content-portion
    tray275 tray164 tray460 - tray
    table450 table265 table276 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray275 kitchen)
     (at tray164 kitchen)
     (at tray460 kitchen)
     (at_kitchen_bread bread423)
     (at_kitchen_content content49)
     (not_allergic_gluten child107)
     (waiting child107 table276)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child107)
    )
  )
)
