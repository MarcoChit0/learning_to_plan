; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 734130

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child253 child468 - child
    bread220 bread249 - bread-portion
    content435 content258 - content-portion
    tray20 tray355 tray330 tray417 - tray
    table271 table110 table77 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray20 kitchen)
     (at tray355 kitchen)
     (at tray330 kitchen)
     (at tray417 kitchen)
     (at_kitchen_bread bread220)
     (at_kitchen_bread bread249)
     (at_kitchen_content content435)
     (at_kitchen_content content258)
     (no_gluten_bread bread220)
     (no_gluten_content content258)
     (allergic_gluten child468)
     (not_allergic_gluten child253)
     (waiting child253 table77)
     (waiting child468 table271)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child253)
     (served child468)
    )
  )
)
