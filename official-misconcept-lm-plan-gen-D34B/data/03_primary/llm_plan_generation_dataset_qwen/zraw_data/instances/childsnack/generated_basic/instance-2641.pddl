; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 473668

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child159 - child
    bread104 - bread-portion
    content61 - content-portion
    tray365 tray311 tray65 tray69 - tray
    table463 table69 table86 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray365 kitchen)
     (at tray311 kitchen)
     (at tray65 kitchen)
     (at tray69 kitchen)
     (at_kitchen_bread bread104)
     (at_kitchen_content content61)
     (not_allergic_gluten child159)
     (waiting child159 table463)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child159)
    )
  )
)
