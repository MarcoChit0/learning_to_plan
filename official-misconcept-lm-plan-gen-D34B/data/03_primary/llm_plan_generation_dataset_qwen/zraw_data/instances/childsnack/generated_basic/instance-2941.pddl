; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 197788

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child177 - child
    bread386 - bread-portion
    content465 - content-portion
    tray256 tray185 tray125 tray92 - tray
    table311 table443 table280 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray256 kitchen)
     (at tray185 kitchen)
     (at tray125 kitchen)
     (at tray92 kitchen)
     (at_kitchen_bread bread386)
     (at_kitchen_content content465)
     (not_allergic_gluten child177)
     (waiting child177 table443)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child177)
    )
  )
)
