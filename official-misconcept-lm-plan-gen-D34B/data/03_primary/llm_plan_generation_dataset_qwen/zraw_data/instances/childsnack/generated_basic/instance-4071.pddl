; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 995998

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child92 - child
    bread162 - bread-portion
    content416 - content-portion
    tray423 tray327 tray220 - tray
    table31 table20 table164 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray423 kitchen)
     (at tray327 kitchen)
     (at tray220 kitchen)
     (at_kitchen_bread bread162)
     (at_kitchen_content content416)
     (not_allergic_gluten child92)
     (waiting child92 table31)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child92)
    )
  )
)
