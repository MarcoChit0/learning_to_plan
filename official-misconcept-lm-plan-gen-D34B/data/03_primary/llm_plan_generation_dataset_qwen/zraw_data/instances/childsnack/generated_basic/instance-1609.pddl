; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 27067

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child165 - child
    bread230 - bread-portion
    content250 - content-portion
    tray246 tray233 tray17 tray355 - tray
    table311 table174 table159 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray246 kitchen)
     (at tray233 kitchen)
     (at tray17 kitchen)
     (at tray355 kitchen)
     (at_kitchen_bread bread230)
     (at_kitchen_content content250)
     (not_allergic_gluten child165)
     (waiting child165 table311)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child165)
    )
  )
)
