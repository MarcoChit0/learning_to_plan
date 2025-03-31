; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 65831

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child219 - child
    bread385 - bread-portion
    content294 - content-portion
    tray185 tray20 tray326 - tray
    table463 table159 table407 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray185 kitchen)
     (at tray20 kitchen)
     (at tray326 kitchen)
     (at_kitchen_bread bread385)
     (at_kitchen_content content294)
     (not_allergic_gluten child219)
     (waiting child219 table407)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child219)
    )
  )
)
